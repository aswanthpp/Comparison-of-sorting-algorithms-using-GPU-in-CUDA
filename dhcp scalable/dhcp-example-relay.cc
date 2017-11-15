#include <fstream>
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"

using namespace ns3;

/*Define a Log component with the name DhcpExample*/
NS_LOG_COMPONENT_DEFINE ("DhcpExampleRelay");

int main (int argc, char *argv[])
{
  bool verbose = false;
  bool tracing = false;
  Time stopTime = Seconds (20);

  CommandLine cmd;
  cmd.AddValue ("verbose", "turn on the logs", verbose);
  cmd.AddValue ("tracing", "turn on the tracing", tracing);
  /*Parse the program arguments*/
  cmd.Parse (argc, argv);

  // GlobalValue::Bind ("ChecksumEnabled", BooleanValue (true));

  if (verbose)
    {
      LogComponentEnable ("DhcpServer", LOG_LEVEL_ALL);
      LogComponentEnable ("DhcpClient", LOG_LEVEL_ALL);
      LogComponentEnable ("DhcpRelay", LOG_LEVEL_ALL);
      LogComponentEnable ("DhcpExampleRelay", LOG_LEVEL_ALL);

    }

  NS_LOG_INFO ("Create nodes.");
  /*NodeContainer - Keeps track of a set of node pointers. Typically ns-3 helpers operate on more than one node at 
  a time. For example, a device helper may want to install devices on a large number of similar nodes. The helper 
  Install methods usually take a NodeContainer as a parameter. NodeContainers hold the multiple Ptr<Node> which 
  are used to refer to the nodes*/
  NodeContainer nodes;
  NodeContainer relay;
  
  /*Create 3 nodes and append pointers to them to the end of this NodeContainer, i.e.,nodes*/
  nodes.Create (3);
  relay.Create (1);

  /*Create a node container which is a concatenation of two input NodeContainers*/
  NodeContainer net (nodes, relay);

  NS_LOG_INFO ("Create channels.");

  /*Build a set of CsmaNetDevice objects*/
  CsmaHelper csma;

  /*SetChannelAttribute - Set these attributes on each ns3::CsmaChannel created by CsmaHelper::Install*/
  csma.SetChannelAttribute ("DataRate", StringValue ("5Mbps"));
  csma.SetChannelAttribute ("Delay", StringValue ("2ms"));

  /*SetDeviceAttribute - Set these attributes on each ns3::CsmaNetDevice created by CsmaHelper::Install*/
  csma.SetDeviceAttribute ("Mtu", UintegerValue (1500));

  /*Install - This method creates an ns3::CsmaChannel with the attributes configured by 
  CsmaHelper::SetChannelAttribute, an ns3::CsmaNetDevice with the attributes configured by 
  CsmaHelper::SetDeviceAttribute and then adds the device to the node, i.e., net and attaches the 
  channel to the device*/
  NetDeviceContainer devNet = csma.Install (net);

  NodeContainer p2pNodes;

  /*Add - Append the contents of another NodeContainer to the end of this container*/
  p2pNodes.Add (net.Get (3));
  p2pNodes.Create (1);

  /*PointToPointHelper - Build a set of PointToPointNetDevice objects*/
  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer p2pDevices;
  p2pDevices = pointToPoint.Install (p2pNodes);

  /*InternetStackHelper - Aggregate IP/TCP/UDP functionality to existing Nodes*/
  InternetStackHelper tcpip;

  /*Install - Aggregate implementations of the ns3::Ipv4, ns3::Ipv6, ns3::Udp, and ns3::Tcp classes onto the 
  provided node*/
  tcpip.Install (nodes);
  tcpip.Install (relay);
  tcpip.Install (p2pNodes.Get (1));

  NS_LOG_INFO ("Setup the IP addresses and create DHCP applications.");
  DhcpHelper dhcpHelper;

   ApplicationContainer dhcpServerApp = dhcpHelper.InstallDhcpServer (p2pDevices.Get (1), Ipv4Address ("172.30.1.12"),
                                                                     Ipv4Address ("172.30.1.0"), Ipv4Mask ("/24"),
                                                                     Ipv4Address ("172.30.1.10"), 
                                                                     Ipv4Address ("172.30.1.16"));
  DynamicCast<DhcpServer> (dhcpServerApp.Get (0))->AddStaticDhcpEntry (devNet.Get (2)->GetAddress (), Ipv4Address ("172.30.0.14"));

  dhcpServerApp.Start (Seconds (0.0));
  dhcpServerApp.Stop (stopTime);

   NetDeviceContainer dhcpRelayNetDevs;
   dhcpRelayNetDevs.Add (devNet.Get (2));
   dhcpRelayNetDevs.Add (p2pDevices.Get(0));

   ApplicationContainer dhcpRelay = dhcpHelper.InstallDhcpRelay (dhcpRelayNetDevs,Ipv4Address ("172.30.1.16"),Ipv4Address ("172.30.0.17")
                                                              ,Ipv4Address("172.30.1.12"),Ipv4Mask ("/24"));
    dhcpRelay.Start (Seconds (1.0));
    dhcpRelay.Stop (stopTime);



  NetDeviceContainer dhcpClientNetDevs;
  dhcpClientNetDevs.Add (devNet.Get (0));
  dhcpClientNetDevs.Add (devNet.Get (1));
  
  ApplicationContainer dhcpClients = dhcpHelper.InstallDhcpClient (dhcpClientNetDevs);
  dhcpClients.Start (Seconds (2.0));
  dhcpClients.Stop (stopTime);


   Simulator::Stop (stopTime + Seconds (10.0));

  if (tracing)
    {
      csma.EnablePcapAll ("dhcp-csma");
      pointToPoint.EnablePcapAll ("dhcp-p2p");
    }

  NS_LOG_INFO ("Run Simulation.");
  Simulator::Run ();
  Simulator::Destroy ();
  NS_LOG_INFO ("Done.");



}

                                                                  



